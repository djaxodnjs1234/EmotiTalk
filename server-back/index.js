const {createServer} = require("http");
const app = require("./app");
const {Server} = require("socket.io");
require("dotenv").config();

const httpServer = createServer(app);
const io = new Server(httpServer, {
    cors: {
        origin: "*",
        credentials: true
    },
});

require("./utils/io")(io);

httpServer.listen(process.env.PORT, () => {
    console.log("server listening on port", process.env.PORT);
});

// require란 JavaScript 파일, 라이브러리, JSON 등 외부 코드를 현재 파일에 포함시키는 역할.
// import 같은 역할, 모듈 불러오기임

