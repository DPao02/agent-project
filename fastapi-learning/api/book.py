from fastapi import APIRouter

api_book = APIRouter()

@api_book.get("/get")
async def read_get():
    return {"method": "get方法"}

@api_book.put("/put")
def read_put():
    return {"method": "put方法"}

@api_book.post("/post")
def read_post():
    return {"method": "post方法"}

@api_book.delete("/delete")
def read_delete():
    return {"method": "delete方法"}