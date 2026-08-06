import gzip

from scripts.download_raw_data import decompress


def test_decompress_writes_uncompressed_content(tmp_path):
    original = "listing_id,date,comments\n1,2024-01-01,hello\n"
    gz_path = tmp_path / "reviews.csv.gz"
    csv_path = tmp_path / "reviews.csv"
    with gzip.open(gz_path, "wt") as f:
        f.write(original)

    decompress(gz_path, csv_path)

    assert csv_path.read_text() == original
