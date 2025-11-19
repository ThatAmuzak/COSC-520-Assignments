import os
import re
import shutil

# ===================================================
# Gutenberg text fetching
# ===================================================
from gutenbergpy.textget import get_text_by_id, strip_headers
from tqdm import tqdm

titles = {
    2701: "Moby-Dick",
    1342: "Pride-and-Prejudice",
    1260: "Jane-Eyre",
    768: "Wuthering-Heights",
    1400: "Great-Expectations",
    345: "Dracula",
    84: "Frankenstein",
    174: "Picture-of-Dorian-Gray",
    120: "Treasure-Island",
    11: "Alice-in-Wonderland",
    12: "Through-the-Looking-Glass",
    829: "Gullivers-Travels",
    98: "A-Tale-of-Two-Cities",
    1661: "Adventures-of-Sherlock-Holmes",
    36: "War-of-the-Worlds",
    121: "Sense-and-Sensibility",
    135: "Les-Miserables",
    2600: "War-and-Peace",
    145: "Middlemarch",
    766: "David-Copperfield",
    1184: "Count-of-Monte-Cristo",
    28054: "The-Brothers-Karamazov",
    1399: "Anna-Karenina",
    2638: "The-Idiot",
    599: "Vanity-Fair",
    2610: "Hunchback-of-Notre-Dame",
    2607: "The-Republic",
    4300: "Ulysses",
    2554: "Crime-and-Punishment",
    219: "The-Secret-Agent",
    1404: "An-American-Tragedy",
    5230: "The-Mysteries-of-Udolpho",
    5232: "The-Italian",
    2805: "The-Man-Who-Was-Thursday",
    244: "The-Jungle",
    3070: "The-Three-Musketeers",
    820: "Uncle-Toms-Cabin",
    2852: "Far-from-the-Madding-Crowd",
    306: "Jude-the-Obscure",
    526: "Tess-of-the-dUrbervilles",
    144: "The-Mill-on-the-Floss",
    550: "The-Return-of-the-Native",
    3618: "The-Memoirs-of-Sherlock-Holmes",
    140: "Works-of-Charles-Dickens-Vol-1",
    141: "Works-of-Charles-Dickens-Vol-2",
    142: "Works-of-Charles-Dickens-Vol-3",
    143: "Works-of-Charles-Dickens-Vol-4",
    4085: "Complete-Works-of-Jack-London",
    10150: "Complete-Works-of-William-Shakespeare",
    8800: "Poetical-Works-of-Longfellow",
    1349: "Outline-of-Science-Vol-1",
    1350: "Outline-of-Science-Vol-2",
    16451: "Story-of-Mankind",
    20417: "Story-of-Philosophy",
    14591: "History-of-England-5-Vols",
    40686: "Decline-and-Fall-of-Roman-Empire-Vol-1",
    731: "Democracy-in-America-Part-1",
    815: "Democracy-in-America-Part-2",
    2591: "The-Histories",
    1414: "The-Pickwick-Papers",
    967: "Mayor-of-Casterbridge",
    3420: "The-Persian-Wars",
    37134: "House-of-the-Seven-Gables",
    2445: "The-Red-and-the-Black",
    521: "Princess-Casamassima",
    5317: "Wings-of-the-Dove",
}

book_ids = list(titles.keys())

root = "data"
books_dir = os.path.join(root, "gutenberg_books")
corpus_path = os.path.join(root, "gutenberg.txt")

os.makedirs(root, exist_ok=True)

# Clear books directory
if os.path.exists(books_dir):
    shutil.rmtree(books_dir)
os.makedirs(books_dir, exist_ok=True)


def remove_empty_lines(s: str) -> str:
    return "\n".join([line for line in s.splitlines() if line.strip()])


print("Generating Gutenberg corpus")

for gid in tqdm(book_ids, desc="Downloading and cleaning"):
    title = titles[gid]
    filename = f"{title}.txt"
    out_path = os.path.join(books_dir, filename)

    # Skip if file already exists
    if os.path.exists(out_path):
        continue

    raw = get_text_by_id(gid)
    clean = strip_headers(raw)
    text = clean.decode("utf-8", errors="ignore")
    text = remove_empty_lines(text)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

with open(corpus_path, "w", encoding="utf-8") as out:
    for gid in book_ids:
        filename = f"{titles[gid]}.txt"
        path = os.path.join(books_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            out.write(f.read() + "\n")

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

characters = len(text)
words = len(re.findall(r"\w+", text))
shutil.rmtree("texts")

print(
    f"Final corpus: {corpus_path}, Character count: {characters}, Word count: {words}"
)
print("==============================")

# ===================================================
# Adversarial text generation
# ===================================================


OUTPUT_PATH = "data/adversarial.txt"
TOTAL_CHARS = 50_000_000  # fifty million
LAST_CHAR = b"b"
CHUNK_SIZE = 1_000_000  # one million bytes per chunk


def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def write_file(path):
    # number of 'a' characters to write
    a_count = TOTAL_CHARS - 1  # 50_000_000 - 1 = 49_999_999
    full_chunks, remainder = divmod(a_count, CHUNK_SIZE)

    ensure_dir(path)
    with open(path, "wb") as f:
        a_chunk = b"a" * CHUNK_SIZE
        for _ in tqdm(range(full_chunks)):
            f.write(a_chunk)
        if remainder:
            f.write(b"a" * remainder)
        # write the final 'b'
        f.write(LAST_CHAR)
        f.flush()
        os.fsync(f.fileno())


print("Generating Adversarial corpus")
write_file(OUTPUT_PATH)
print(f"Final Corpus: {OUTPUT_PATH}, Character Count {TOTAL_CHARS}")
print("==============================")
