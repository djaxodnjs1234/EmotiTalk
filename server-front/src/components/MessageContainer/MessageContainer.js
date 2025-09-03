// src/components/MessageContainer/MessageContainer.js

import React, { useEffect, useRef } from "react";
import "./MessageContainer.css";

const emotionColors = {
  '😄': '#ffe5b4', // 행복
  '🙂': '#ffe5b4',
  '😢': '#cfe2ff', // 슬픔
  '😞': '#cfe2ff',
  '😡': '#f8d7da', // 분노
  '😤': '#f8d7da',
  '😱': '#e2d4f8', // 공포
  '😨': '#e2d4f8',
  '😲': '#fff3cd', // 놀람
  '😯': '#fff3cd',
  '🤢': '#dff1acff', // 혐오
  '😒': '#dff1acff',
  '': '#e9ecef',   // 중립 (emoji가 없는 경우)
};

const MessageContainer = ({ messageList, user, showEmoticon = true }) => {
  const messagesEndRef = useRef(null); // ✅ 스크롤 위치를 참조할 ref

  // 메시지 목록이 업데이트될 때마다 스크롤을 맨 아래로 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messageList]);

  return (
    <div className="chat-wrapper">
      {messageList.map((message, index) => {
        const key = message._id || `message-${index}-${message.chat.substring(0, 5)}`;

        const isSystem = message.user.name === "system";
        const isMyMessage = user && message.user.name === user.name;
        const isFirstMessage = index === 0;
        const prevMessage = messageList[index - 1];
        const isSameSenderAsPrev =
          !isFirstMessage && prevMessage.user.name === message.user.name;

        // ✅ 감정별 색상 스타일을 동적으로 적용합니다.
        const dynamicStyle = showEmoticon
          ? { backgroundColor: emotionColors[message.emoticon] || emotionColors[''] } // On 상태: 감정 색상 적용
          : {backgroundColor: emotionColors['']}; 

        // 이모티콘 출력 여부 설정
        const emoji = showEmoticon && message.emoticon ? (
          <span style={{ marginLeft: '8px', fontSize: '1.5rem' }}>{message.emoticon}</span>
      ) : null;

        return (
          <div
            key={key}
            className={`message-item-wrapper ${isSystem ? "system" : isMyMessage ? "my" : "your"}`}
          >
            {isSystem ? (
              <div className="system-message-container">
                <p className="system-message">{message.chat}</p>
              </div>
            ) : isMyMessage ? (
              <div className="my-message-container">
                {/* ✅ 동적으로 생성한 style을 적용 */}
                <div className="my-message" style={dynamicStyle}>
                  {/* 내 메시지 + 이모티콘 */}
                  <span>{message.chat}</span>
                  {emoji}
                </div>
              </div>
            ) : (
              <div className="your-message-container">
                {/* 프로필 이미지 (이전 보낸 사람과 다를 때만 표시) */}
                {!isSameSenderAsPrev ? (
                  <img
                    src="/profile.jpeg"
                    alt="profile"
                    className="profile-image"
                  />
                ) : (
                  <div className="profile-placeholder"></div>
                )}

                <div className="your-message-content">
                  {/* 유저 이름 (이전 보낸 사람과 다를 때만 표시) */}
                  {!isSameSenderAsPrev && (
                    <div className="user-name">{message.user.name}</div>
                  )}
                  {/* ✅ 동적으로 생성한 style을 적용 */}
                  <div className="your-message" style={dynamicStyle}>
                    {/* 상대 메시지 + 이모티콘 */}
                    <span>{message.chat}</span>
                    {emoji}
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}
      {/* 스크롤 제어용 빈 요소 */}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageContainer;