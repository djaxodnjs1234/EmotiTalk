//Controllers/user.controller.js
const User = require("../models/user");
const userController = {}

// 회원가입
userController.registerUser = async (userName, password, sid) => {
    let user = await User.findOne({ name: userName });
    if (user) throw new Error("이미 존재하는 사용자명입니다.");
    user = new User({
        name: userName,
        password,
        token: sid,
        online: true,
    });
    await user.save();
    return user;
};

// 로그인
userController.loginUser = async (userName, password, sid) => {
    const user = await User.findOne({ name: userName });
    if (!user) throw new Error("존재하지 않는 사용자입니다.");
    const correct = await user.comparePassword(password);
    if (!correct) throw new Error("비밀번호가 틀렸습니다.");
    user.token = sid;
    user.online = true;
    await user.save();
    return user;
};

userController.checkUser = async (sid) => {
    const user = await User.findOne({ token: sid });
    if (!user) throw new Error("user not found");
    return user;
};

userController.saveUser = async(userName, sid)=>{
    //이미 있는 유저인지 확인
    let user = await User.findOne({ name: userName});

    //없다면 새로운 유저정보 만들기
    if(!user){
        user = new User({
            name: userName,
            token: sid,
            online: true,
        });
    }

    //이미 있는 유저라면 연결정보 토큰값만 바꿔주기
    user.token = sid;
    user.online = true;

    await user.save();
    return user;
};

userController.checkUser = async(sid)=>{
    const user = await User.findOne({token:sid})
    if(!user)throw new Error("user not found");
    return user;
}

module.exports = userController;