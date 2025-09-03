// routes/auth.js
const express = require("express");
const router = express.Router();
const userController = require("../Controllers/user.controller");

// 회원가입
router.post("/register", async (req, res) => {
    try {
        const { name, password } = req.body;
        const user = await userController.registerUser(name, password, req.body.sid || "");
        res.status(201).json({ ok: true, user });
    } catch (error) {
        res.status(400).json({ ok: false, error: error.message });
    }
});

// 로그인
router.post("/login", async (req, res) => {
    try {
        const { name, password } = req.body;
        const user = await userController.loginUser(name, password, req.body.sid || "");
        res.status(200).json({ ok: true, user });
    } catch (error) {
        res.status(400).json({ ok: false, error: error.message });
    }
});

module.exports = router;