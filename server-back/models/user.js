const mongoose = require("mongoose");
const bcrypt = require("bcrypt");

const userSchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, "User must type name"],
        unique: true,
    },
    password: {
        type: String,
        required: [true, "User must type password"]
    },
    token: {
        type: String, // 연결 ID 정보
    },
    online: {
        type: Boolean,
        default: false,
    },
});

// 비밀번호 해시 저장
userSchema.pre("save", async function (next) {
    if (this.isModified("password")) {
        this.password = await bcrypt.hash(this.password, 10);
    }
    next();
});

userSchema.methods.comparePassword = function (candidatePassword) {
    return bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.model("User", userSchema);