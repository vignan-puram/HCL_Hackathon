"""FastAPI endpoint integration and contract tests."""

import io
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "index" in data
    assert "documents" in data["index"]
    assert "chunks" in data["index"]
    assert "llm_provider" in data


def test_documents_endpoint():
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)


def test_query_validation():
    # Empty question
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "INVALID_REQUEST"


def test_missing_document_returns_404():
    response = client.get("/documents/nonexistent_doc_id/file")
    assert response.status_code == 404
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_missing_chunks_returns_404():
    response = client.get("/documents/nonexistent_doc_id/chunks")
    assert response.status_code == 404
    err = response.json()
    assert err["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_upload_unsupported_format():
    file_content = b"Binary data or unknown file"
    files = [("files", ("unsupported.exe", io.BytesIO(file_content), "application/octet-stream"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "failed"
    assert data["results"][0]["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_upload_and_delete_lifecycle():
    sample_txt = b"Enterprise Security Policy\n\nAll staff must use multi-factor authentication (MFA)."
    files = [("files", ("test_security_policy.txt", io.BytesIO(sample_txt), "text/plain"))]
    
    # 1. Upload
    upload_res = client.post("/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert len(upload_data["results"]) == 1
    item = upload_data["results"][0]
    assert item["status"] in ("indexed", "duplicate")
    doc_id = item["document"]["doc_id"]
    
    # 2. Check file retrieval
    file_res = client.get(f"/documents/{doc_id}/file")
    assert file_res.status_code == 200
    assert b"multi-factor authentication" in file_res.content

    # 3. Check chunks inspection
    chunks_res = client.get(f"/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) > 0
    assert chunks[0]["doc_id"] == doc_id

    # 4. Query against the uploaded doc
    query_res = client.post("/query", json={"question": "What authentication must all staff use?"})
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "status" in q_data
    assert "answer" in q_data
    assert "citations" in q_data

    # 5. Delete document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200

    # 6. Verify 404 after deletion
    file_res_after = client.get(f"/documents/{doc_id}/file")
    assert file_res_after.status_code == 404

