---
title: '关于Cursor中Python代码的高亮问题'
date: '2024-12-05'
author: 'Hai'
coverImage: 'https://global.discourse-cdn.com/flex020/uploads/cursor1/optimized/2X/0/045bbffe7a10c30d5b883b0a16862a446cb65c2f_2_1035x456.png'
tags: ['cursor', 'python', 'highlighting']
status: 'published'
---



# 关于Cursor中Python代码的高亮问题

Cursor是一款AI驱动的代码编辑器，支持多种编程语言，包括Python。为了更好地在Cursor中使用Python，我们可以通过修改配置来使用微软的扩展市场，并安装相关Python扩展以实现代码高亮等功能。

![Cursor默认显示python代码样式](https://global.discourse-cdn.com/flex020/uploads/cursor1/optimized/2X/0/045bbffe7a10c30d5b883b0a16862a446cb65c2f_2_1035x456.png)
*Cursor默认显示python代码样式*


以下是详细的设置步骤：

## 退出Cursor并移除所有扩展
1. 在Cursor中，首先卸载所有已安装的扩展。
2. 退出Cursor编辑器。

## 修改Cursor配置文件
根据你的操作系统，找到Cursor的`product.json`文件并进行编辑。
- **MacOS**:
  ```
  /Applications/Cursor.app/Contents/Resources/app/product.json
  ```
- **Windows**:
  ```
  C:\Users\<用户名>\AppData\Local\Programs\cursor\resources\app\product.json
  ```
- **Linux**:
  ```
  /usr/lib/code/product.json
  ```
打开该文件后，找到`extensionsGallery`键，并按照以下内容进行修改：
原`extensionsGallery`内容：
```json
{
    "galleryId": "cursor",
    "serviceUrl": "https://marketplace.cursorapi.com/_apis/public/gallery",
    "itemUrl": "https://marketplace.cursorapi.com/items",
    "resourceUrlTemplate": "https://marketplace.cursorapi.com/{publisher}/{name}/{version}/{path}",
    "controlUrl": "",
    "recommendationsUrl": "",
    "nlsBaseUrl": "",
    "publisherUrl": ""
}
```
修改后的`extensionsGallery`内容：
```json
{
    "galleryId": "cursor",
    "serviceUrl": "https://marketplace.visualstudio.com/_apis/public/gallery",
    "itemUrl": "https://marketplace.visualstudio.com/items",
    "resourceUrlTemplate": "https://{publisher}.vscode-unpkg.net/{publisher}/{name}/{version}/{path}",
    "controlUrl": "",
    "recommendationsUrl": "",
    "nlsBaseUrl": "",
    "publisherUrl": ""
}
```

## 重新打开Cursor并安装Python扩展

1. 保存对`product.json`文件的修改，并重新打开Cursor。
2. 在扩展市场中搜索并安装以下扩展：
   - `ms-python.python`
   - `ms-python.vscode-pylance`
   - `ms-python.debugpy`

## 确认Python语言服务器设置

确保在Cursor的设置中，`python.languageServer`的值被设置为`"Pylance"`。
完成以上步骤后，你的Cursor编辑器应该已经支持Python代码的高亮，并且可以使用Pylance作为语言服务器，提供更加丰富的Python编程体验。

---
参考链接：[GitHub Gist](https://gist.github.com/joeblackwaslike/752b26ce92e3699084e1ecfc790f74b2#file-readme-md)
