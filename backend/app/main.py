from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import recommendations, stream, health
from app.core.config import settings
from app.services.s3_service import S3Service


@asynccontextmanager
async def lifespan(app: FastAPI):
	app.state.s3_service = S3Service()
	yield


app = FastAPI(
	title="Mess-AI Classical RecSys API",
	version="0.1.0",
	docs_url="/docs",
	redoc_url="/redoc",
	lifespan=lifespan
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000","https://mess-ai.vercel.app"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


app.include_router(stream.router, tags = ["streaming"])
app.include_router(recommendations.router, tags = ["recommendations"])
app.include_router(health.router, tags= ["health"])



if __name__ == '__main__':
	import uvicorn
	uvicorn.run("app.main:app", host='0.0.0.0', port=8000, reload=True)
