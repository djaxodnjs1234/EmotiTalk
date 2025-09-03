import React from "react";
import "./UserList.css";

function UserList({ userList }) {
  return (
    <div className="user-list-area">
      <h4>접속자 목록</h4>
      <ul>
        {userList.map(u => (
          <li key={u.id}>{u.name}</li>
        ))}
      </ul>
    </div>
  );
}

export default UserList;