// import React from "react";

// function RoomList({ rooms, currentRoom, onSelectRoom }) {
//   return (
//     <div className="room-list-area">
//       <h4>채팅방 목록</h4>
//       <ul>
//         {rooms.map((room) => (
//           <li
//             key={room}
//             onClick={() => onSelectRoom(room)}
//             className={room === currentRoom ? "active" : ""}
//             style={{ cursor: "pointer", padding: "8px", borderRadius: "6px", marginBottom: "6px" }}
//           >
//             {room}
//           </li>
//         ))}
//       </ul>
//     </div>
//   );
// }

// export default RoomList;