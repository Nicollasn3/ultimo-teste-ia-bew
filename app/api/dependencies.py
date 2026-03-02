from fastapi import HTTPException, Request


def get_retriever(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever não inicializado")
    return retriever
