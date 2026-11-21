from fastapi import APIRouter

api_zz = APIRouter()

@api_zz.get("/get")
async def read_get():
    return {"method": "get方法"}

@api_zz.put("/put")
def read_put():
    return {"method": "put方法"}

@api_zz.post("/post")
def read_post():
    return {"method": "post方法"}

@api_zz.delete("/delete")
def read_delete():
    return {"method": "delete方法"}