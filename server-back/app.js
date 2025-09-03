
require('dotenv').config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const app = express();
const corsOptions = {
  origin: "*",
  credentials: true, 
};

app.use(cors(corsOptions));
app.use(express.json()); // JSON 요청을 파싱하기 위해 추가

// 1. chat.js 라우터 불러오기
const chatRoutes = require("./routes/chat");
const authRoutes = require("./routes/auth"); // 추가

//3. Db 연결
mongoose.connect(process.env.DB)
  .then(() => console.log("✅ DB 연결 성공"))
  .catch((err) => console.error("❌ DB 연결 실패:", err));

// 3. /api/chats 경로에 라우터 등록
app.use("/api/chats", chatRoutes);
// 4. /api/auth 경로에 라우터 등록
app.use("/api/auth", authRoutes);

module.exports = app;