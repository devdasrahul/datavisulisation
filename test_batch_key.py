from ingest.ingest_batches import make_batch_key


def test_make_batch_key_format():
    assert make_batch_key("FD001", unit_id=7, batch_idx=12) == "FD001_unit0007_batch000012"
