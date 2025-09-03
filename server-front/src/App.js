import { useEffect, useState } from "react";
import "./App.css";
import socket from "./server";
import Draggable from "react-draggable";
import InputField from "./components/InputField/InputField";
import MessageContainer from "./components/MessageContainer/MessageContainer";
import LoginModal from "./components/LoginModal/LoginModal";
import UserList from "./components/UserList/UserList";

function App() {
  const [user, setUser] = useState(null);
  const [message, setMessage] = useState("");
  const [messageList, setMessageList] = useState([]);
  const [userList, setUserList] = useState([]);
  
  // 이모티콘 On/Off 상태 (기본: On)
  const [showEmoticon, setShowEmoticon] = useState(true);

  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // DB에서 과거 메시지 불러오기
    const fetchInitialMessages = async () => {
      try {
        const response = await fetch("/api/chats");
        if (!response.ok) {
          throw new Error("과거 대화 기록을 불러오는데 실패했습니다.");
        }
        const data = await response.json();
        setMessageList(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchInitialMessages();

    // 실시간 유저 목록 수신
    socket.on("userList", setUserList);
    
    // 실시간 메시지 수신 처리
    const handleMessage = (newMessage) => {
      setMessageList((prevState) => [...prevState, newMessage]);
    };
    socket.on("message", handleMessage);

    return () => {
      socket.off("userList");
      socket.off("message", handleMessage);
    };
  }, []);

  // 로그인
  const handleLogin = ({ name, password }) => {
    return new Promise((resolve, reject) => {
      socket.emit("login", { name, password }, (res) => {
        if (res?.ok) {
          setUser(res.user || res.data);
          resolve(res.user || res.data);
        } else {
          reject(new Error(res.error || "로그인 실패"));
        }
      });
    });
  };

  // 회원가입
  const handleRegister = ({ name, password }) => {
    return new Promise((resolve, reject) => {
      socket.emit("register", { name, password }, (res) => {
        if (res?.ok) {
          resolve(res.user || res.data);
        } else {
          reject(new Error(res.error || "회원가입 실패"));
        }
      });
    });
  };

  // 메시지 전송
  const sendMessage = (event) => {
    event.preventDefault();
    if (!message.trim()) return;
    
    // ✅ Flask API 호출 로직을 제거하고, Node.js 백엔드에서 모든 감정 분석을 처리하도록 변경합니다.
    // Node.js 백엔드에 메시지 문자열만 보냅니다.
    socket.emit("sendMessage", message, (res) => {
      if (res?.ok) {
        setMessage("");
      } else {
        console.error("메시지 전송 실패:", res.error);
        alert(`메시지 전송 실패: ${res.error}`);
      }
    });
  };

  const toggleEmoticon = () => {
    setShowEmoticon((prev) => !prev);
  };

  return (
    <div className={`App ${darkMode ? "dark" : ""}`}>
      {!user && <LoginModal onLogin={handleLogin} onRegister={handleRegister} />}
      {user && (
        <>
          <Draggable bounds="body">
            <div style={{ position: "fixed", top: 50, right: 20, zIndex: 100 }}>
              <UserList userList={userList} />
            </div>
          </Draggable>

          {/* ✅ 다크모드 토글 버튼 */}
          <button
            onClick={() => setDarkMode((prev) => !prev)}
            style={{
              position: "fixed",
              top: 10,
              right: 20,
              zIndex: 200,
              padding: "5px 10px",
              borderRadius: "8px",
              border: "none",
              cursor: "pointer",
            }}
          >
            {darkMode ? "☀️ 라이트모드" : "🌙 다크모드"}
          </button>
          <MessageContainer
            messageList={messageList}
            user={user}
            showEmoticon={showEmoticon}
          />
          <InputField
            message={message}
            setMessage={setMessage}
            sendMessage={sendMessage}
            showEmoticon={showEmoticon}
            toggleEmoticon={() => setShowEmoticon(prev => !prev)}
          />
        </>
      )}
    </div>
  );
}

export default App;