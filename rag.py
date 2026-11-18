"""Batch retrieval and grounded answer generation for the RAG case."""

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

LLM = "t-tech/T-lite-it-2.1"
EMBEDDER = "ai-sage/Giga-Embeddings-instruct"
RERANKER = "BAAI/bge-reranker-v2-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500
EMBEDDING_BLOCK = 256
RETRIEVAL_TOP_K = 30
CONTEXT_TOP_K = 5
CONTEXT_TOKENS = 3200
NO_ANSWER = "Данных нет"
HYDE_PROMPT = (
    "Ты — эксперт. Пользователь задает вопрос. "
    "Напиши краткий, логичный и вероятный ответ на этот вопрос. "
    "Пиши только сухие факты, которые могут содержаться в идеальном документе-ответе. "
    "Не используй вводных слов."
)
ANSWER_PROMPT = (
    "Ты отвечаешь на вопросы об Альфа-Банке, используя ТОЛЬКО предоставленный контекст. "
    "Отвечай максимально кратко и ёмко, без приветствий и вводных слов. "
    "Пиши только факты. Если в контексте нет прямого ответа на вопрос, "
    f"выдай фразу '{NO_ANSWER}'."
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/questions.csv"))
    parser.add_argument("--websites", type=Path, default=Path("data/websites.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".rag-cache"))
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--hyde-batch-size", type=positive_int, default=16)
    parser.add_argument("--embedding-batch-size", type=positive_int, default=8)
    parser.add_argument("--search-batch-size", type=positive_int, default=256)
    parser.add_argument("--rerank-batch-size", type=positive_int, default=8)
    parser.add_argument("--generation-batch-size", type=positive_int, default=64)
    return parser.parse_args()


def read_inputs(questions_path, websites_path):
    questions = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    websites = pd.read_csv(websites_path, dtype=str, keep_default_na=False)
    if not {"q_id", "query"}.issubset(questions.columns):
        raise ValueError("questions.csv must contain q_id and query")
    if questions.empty or questions.q_id.str.strip().eq("").any() or not questions.q_id.is_unique:
        raise ValueError("questions.csv must contain nonempty, unique q_id values")
    if questions["query"].str.strip().eq("").any():
        raise ValueError("questions.csv contains empty queries")
    # The notebook used text, while the case description calls the column web.
    text_column = next((name for name in ("web", "text") if name in websites), None)
    if text_column is None:
        raise ValueError("websites.csv must contain a web or text column")
    return questions.q_id.tolist(), questions["query"].tolist(), websites[text_column].tolist()


def build_chunks(documents):
    chunks = []
    for document in documents:
        for start in range(0, len(document), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = document[start:start + CHUNK_SIZE].strip()
            if len(chunk) > 30:
                chunks.append(chunk)
    if not chunks:
        raise ValueError("No usable chunks in websites.csv")
    return chunks


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_cache(args):
    inputs = [
        file_digest(args.questions),
        file_digest(args.websites),
        file_digest(Path(__file__)),
        args.attention,
    ]
    cache_key = hashlib.sha256("\n".join(map(str, inputs)).encode()).hexdigest()
    cache = args.cache_dir / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, writer):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, lambda target: target.write_text(
        json.dumps(value, ensure_ascii=False), encoding="utf-8"
    ))


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)
    tokenizer.padding_side = tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_to_cache(requests, system, path, tokenizer, args, *, hyde=False):
    from transformers import AutoModelForCausalLM

    results = read_json(path) if path.is_file() else []
    if len(results) == len(requests):
        return results

    batch_size = args.hyde_batch_size if hyde else args.generation_batch_size
    max_length, max_new_tokens = (1024, 150) if hyde else (4096, 120)
    sampling = {"do_sample": True, "temperature": 0.3} if hyde else {"do_sample": False}
    model = AutoModelForCausalLM.from_pretrained(
        LLM, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attention, trust_remote_code=True,
    ).eval()

    for start in tqdm(range(len(results), len(requests), batch_size), desc=path.stem):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": request}],
                tokenize=False, add_generation_prompt=True,
            )
            for request in requests[start:start + batch_size]
        ]
        inputs = tokenizer(
            prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, **sampling,
            )
        generated = outputs[:, inputs.input_ids.shape[1]:]
        results.extend(text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True))
        save_json(path, results)
        del inputs, outputs, generated
    return results


def embed(chunks, expanded_queries, cache, args):
    from sentence_transformers import SentenceTransformer

    blocks = [
        (start, cache / f"docs_{start:08d}.pt")
        for start in range(0, len(chunks), EMBEDDING_BLOCK)
    ]
    query_path = cache / "queries.pt"
    if not query_path.is_file() or any(not path.is_file() for _, path in blocks):
        model = SentenceTransformer(
            EMBEDDER, device="cuda", trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16, "attn_implementation": args.attention},
        )
        model.max_seq_length = 512
        options = dict(
            batch_size=args.embedding_batch_size, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        for start, path in tqdm(blocks, desc="Document embeddings"):
            if not path.is_file():
                vectors = model.encode_document(chunks[start:start + EMBEDDING_BLOCK], **options)
                vectors = vectors.cpu().to(torch.float16).contiguous()
                atomic_write(path, lambda target: torch.save(vectors, target))
        if not query_path.is_file():
            vectors = model.encode_query(expanded_queries, **options)
            vectors = vectors.cpu().to(torch.float16).contiguous()
            atomic_write(query_path, lambda target: torch.save(vectors, target))
        del model
        release_gpu()

    documents = torch.cat([torch.load(path, map_location="cpu", weights_only=True) for _, path in blocks])
    queries = torch.load(query_path, map_location="cpu", weights_only=True)
    return documents, queries


def tokenize(text):
    return re.findall(r"(?u)\b\w+\b", text.lower())


def minmax(scores):
    low, high = scores.min(), scores.max()
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


def hybrid_search(chunks, queries, cache, args):
    path = cache / "candidates.json"
    if path.is_file():
        return read_json(path)

    expanded = [f"query: {query}\n{answer}" for query, answer in zip(queries, read_json(cache / "hyde.json"))]
    docs, query_vectors = embed(chunks, expanded, cache, args)
    docs = docs.to(device="cuda", dtype=torch.float32)
    query_vectors = query_vectors.to(device="cuda", dtype=torch.float32)
    bm25 = BM25Okapi([tokenize(chunk) for chunk in chunks])
    candidates = []
    for start in tqdm(range(0, len(queries), args.search_batch_size), desc="Hybrid search"):
        dense_batch = (query_vectors[start:start + args.search_batch_size] @ docs.T).cpu().numpy()
        for offset, dense in enumerate(dense_batch):
            sparse = np.asarray(bm25.get_scores(tokenize(queries[start + offset])), dtype=np.float32)
            scores = 0.7 * minmax(dense) + 0.3 * minmax(sparse)
            candidates.append(np.argsort(scores)[::-1][:RETRIEVAL_TOP_K].tolist())
    save_json(path, candidates)
    return candidates


def build_context(chunks, ranking, tokenizer):
    selected, used_tokens = [], 0
    for index in ranking[:CONTEXT_TOP_K]:
        chunk = chunks[index]
        tokens = tokenizer(chunk, add_special_tokens=False).input_ids
        if not selected and len(tokens) > CONTEXT_TOKENS:
            return tokenizer.decode(tokens[:CONTEXT_TOKENS], skip_special_tokens=True)
        if used_tokens + len(tokens) > CONTEXT_TOKENS:
            break
        selected.append(chunk)
        used_tokens += len(tokens)
    return "\n---\n".join(selected)


def rerank(chunks, queries, candidates, cache, tokenizer, args):
    from sentence_transformers import CrossEncoder

    path = cache / "contexts.json"
    contexts = read_json(path) if path.is_file() else []
    if len(contexts) == len(queries):
        return contexts
    model = CrossEncoder(RERANKER, device="cuda", max_length=512, model_kwargs={"dtype": torch.bfloat16})
    for index in tqdm(range(len(contexts), len(queries)), desc="Reranking"):
        indices = candidates[index]
        pairs = [(queries[index], chunks[candidate]) for candidate in indices]
        scores = model.predict(pairs, batch_size=args.rerank_batch_size, show_progress_bar=False)
        ordering = np.argsort(np.asarray(scores).reshape(-1))[::-1]
        contexts.append(build_context(chunks, [indices[i] for i in ordering], tokenizer))
        save_json(path, contexts)
    return contexts


def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("This pipeline requires a CUDA GPU with bfloat16 support.")
    if args.output.resolve() in {args.questions.resolve(), args.websites.resolve()}:
        raise ValueError("--output must be different from the input files")
    ids, queries, documents = read_inputs(args.questions, args.websites)
    cache = run_cache(args)
    print(f"Checkpoints: {cache}")
    chunks = build_chunks(documents)
    del documents
    tokenizer = load_tokenizer()

    if not (cache / "candidates.json").is_file():
        generate_to_cache(queries, HYDE_PROMPT, cache / "hyde.json", tokenizer, args, hyde=True)
        release_gpu()
    candidates = hybrid_search(chunks, queries, cache, args)
    release_gpu()
    contexts = rerank(chunks, queries, candidates, cache, tokenizer, args)
    release_gpu()

    requests = [f"Контекст:\n{context}\n\nВопрос: {query}\nОтвет:" for query, context in zip(queries, contexts)]
    answers = generate_to_cache(requests, ANSWER_PROMPT, cache / "answers.json", tokenizer, args)
    if len(answers) != len(ids):
        raise RuntimeError("The number of generated answers differs from the number of questions")
    output = pd.DataFrame({"q_id": ids, "answer": [answer or NO_ANSWER for answer in answers]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, lambda target: output.to_csv(target, index=False))
    print(f"{len(output)} answers saved to {args.output}")


if __name__ == "__main__":
    main()
