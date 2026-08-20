from arag.ingestion import _parse_atom


def test_atom_parser_extracts_expected_fields(arxiv_xml: str):
    docs = _parse_atom(arxiv_xml)
    assert len(docs) == 1
    d = docs[0]
    assert d.arxiv_id == "1234.56789v1"
    assert d.title == "Testing the arXiv parser"
    assert d.authors == ["Test Author"]
    assert "cs.CL" in d.categories
    assert d.pdf_url == "http://arxiv.org/pdf/1234.56789v1"
    assert d.published.startswith("2024-01-01")
    assert d.abstract.startswith("A short abstract")


def test_atom_parser_handles_empty_feed():
    xml = (
        '<?xml version="1.0"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom"'
        ' xmlns:arxiv="http://arxiv.org/schemas/atom"></feed>'
    )
    assert _parse_atom(xml) == []


def test_atom_parser_falls_back_to_constructed_pdf_url():
    xml = (
        '<?xml version="1.0"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom"'
        ' xmlns:arxiv="http://arxiv.org/schemas/atom">'
        '<entry>'
        '<id>http://arxiv.org/abs/9999.99999v2</id>'
        '<title>t</title><summary>s</summary>'
        '<author><name>A</name></author>'
        '<arxiv:primary_category term="cs.AI"/>'
        '<published>2024-01-01T00:00:00Z</published>'
        '</entry></feed>'
    )
    d = _parse_atom(xml)[0]
    # Missing pdf link -> constructed URL using version-stripped id
    assert d.pdf_url.endswith("9999.99999")
