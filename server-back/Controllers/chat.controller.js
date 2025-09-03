//Controllers/chat.controller.js
const chat = require("../models/chat");
const chatController = {};

chatController.saveChat = async(message, user, emoticon, topEmotion)=>{
    
    const newMessage = new chat({
        chat:message,
        user:{
            id:user._id,
            name:user.name
        },
        
        emoticon: emoticon,
    });

    await newMessage.save();
    return newMessage;
};

module.exports=chatController;
