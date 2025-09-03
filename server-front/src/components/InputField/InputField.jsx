// InputField.jsx

import React from 'react';
import { Input } from "@mui/base/Input";
import { Button } from "@mui/base/Button";
import './InputField.css';

const InputField = ({ message, setMessage, sendMessage, showEmoticon, toggleEmoticon }) => {
  return (
    <div className="input-area">
      {/* 이모티콘 토글 버튼 */}
      <div 
        className="plus-button"
        onClick={toggleEmoticon}
        title="이모티콘 On/Off"
      >
        {showEmoticon ? "😊" : "＋"}
      </div>

      <form onSubmit={sendMessage} className="input-container">
        <Input
          placeholder="Type in here…"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          multiline={false}
          rows={1}
        />

        <Button
          disabled={message === ""}
          type="submit"
          className="send-button"
        >
          전송
        </Button>
      </form>
    </div>
  );
};

export default InputField;