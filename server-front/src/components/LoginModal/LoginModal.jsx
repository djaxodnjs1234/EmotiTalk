import React, { useState } from "react";
import "./LoginModal.css";

const LoginModal = ({ onLogin, onRegister }) => {
  const [mode, setMode] = useState("login"); // login or register
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);

  const isValid = name.trim() !== "" && password.trim() !== "";

  const resetForm = () => {
    setName("");
    setPassword("");
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isValid) return;
    setError(null);

    try {
      if (mode === "login") {
        await onLogin({ name, password });
      } else {
        await onRegister({ name, password });
        setMode("login"); // 회원가입 완료 후 로그인 탭으로 전환
        resetForm();
        alert("회원가입이 완료되었습니다. 로그인해주세요.");
      }
    } catch (err) {
      setError(err.message || "오류가 발생했습니다.");
    }
  };

  return (
    <div className="login-modal-overlay">
      <form className="login-modal" onSubmit={handleSubmit}>
        <h2>{mode === "login" ? "로그인" : "회원가입"}</h2>

        <div className="tab-buttons">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => { setMode("login"); resetForm(); }}
          >
            로그인
          </button>
          <button
            type="button"
            className={mode === "register" ? "active" : ""}
            onClick={() => { setMode("register"); resetForm(); }}
          >
            회원가입
          </button>
        </div>

        {error && <div className="login-error">{error}</div>}

        <input
          type="text"
          placeholder="이름을 입력하세요"
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoFocus
        />
        <input
          type="password"
          placeholder="비밀번호를 입력하세요"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit" disabled={!isValid}>
          {mode === "login" ? "로그인" : "회원가입"}
        </button>
      </form>
    </div>
  );
};

export default LoginModal;