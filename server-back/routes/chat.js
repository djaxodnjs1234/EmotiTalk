const express = require("express");
const router = express.Router();
const Chat = require("../models/chat"); // chat 모델 경로 확인

// GET /api/chats - DB에 저장된 채팅 내역을 가져오는 API 엔드포인트
router.get("/", async (req, res) => {
  try {
    // 생성된 시간 순서대로, 최근 100개 메시지만 가져오기
    const messages = await Chat.find().sort({ createdAt: 1 }).limit(100);
    res.status(200).json(messages);
  } catch (error) {
    console.error("채팅 내역 조회 중 오류:", error);
    res.status(500).json({ error: "서버에서 오류가 발생했습니다." });
  }
});

module.exports = router;