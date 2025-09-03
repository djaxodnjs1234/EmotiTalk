// models/chat.js
const mongoose = require("mongoose");

const chatSchema = new mongoose.Schema(
    {
        chat: String,
        user: {
            id: {
                type: mongoose.Schema.ObjectId,
                ref: "User",
            },
            name: String,
        },
        emoticon: {
            type: String,
            default: ''
        },
        topEmotion: {  // ✅ 감정 결과 저장
            type: String,
            default: '중립'
        }
    },
    { timestamps: true }
);

module.exports = mongoose.model("Chat", chatSchema);
