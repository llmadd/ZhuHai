export const i18n = {
    zh: {
        nav: {
            home: '首页',
            posts: '文章',
            about: '关于'
        },
        post: {
            publishedAt: '发布于',
            author: '作者',
            tags: '标签',
            share: '分享文章',
            noContent: '暂无内容',
            tableOfContents: '目录',
            backToPostList: '返回文章列表',
            scanQRCode: '扫描二维码查看文章',
            generating: '生成中...',
            downloadShareImage: '下载分享图'
        },
        links: {
            friendLinks: '友情链接'
        },
        home: {
            latestPosts: '最新文章',
            readMore: '阅读更多',
            noPosts: '暂无文章'
        },
        category: {
            title: '博客文章',
            description: '分享技术文章和生活感悟',
            allCategories: '全部',
            noPostsInCategory: '该分类下暂无文章'
        },
        profile: {
            name: '🐼 Hai',
            location: '📍 上海',
            email: '✉️ zh@useai.cn',
            wechat: '💬 微信',
            viewBlog: '👀 查看博客',
            wechatQRCode: '👀 微信二维码',
            description: `目前从事大模型算法工程师，会写一些自己遇到的问题和解决方案。
ps: 对不起，我的文章AI成分很浓，阅读请谨慎！（自己写文章太累啦）
🐼菜鸟闯天涯，欢迎交流！`
        },
        search: {
            placeholder: '搜索文章...',
            noResults: '未找到相关文章',
            searching: '搜索中...'
        }
    },
    en: {
        nav: {
            home: 'Home',
            posts: 'Posts',
            about: 'About'
        },
        post: {
            publishedAt: 'Published at',
            author: 'Author',
            tags: 'Tags',
            share: 'Share',
            noContent: 'Content not available',
            tableOfContents: 'Table of Contents',
            backToPostList: 'Back to Post List',
            scanQRCode: 'Scan QR Code to View Article',
            generating: 'Generating...',
            downloadShareImage: 'Download Share Image'
        },
        links: {
            friendLinks: 'Friend Links'
        },
        home: {
            latestPosts: 'Latest Posts',
            readMore: 'Read More',
            noPosts: 'No posts yet'
        },
        category: {
            title: 'Blog Posts',
            description: 'Sharing technical articles and life insights',
            allCategories: 'All',
            noPostsInCategory: 'No posts in this category'
        },
        profile: {
            name: '🐼 Hai',
            location: '📍 Shanghai',
            email: '✉️ zh@useai.cn',
            wechat: '💬 WeChat',
            viewBlog: '👀 View Blog',
            wechatQRCode: '👀 WeChat QR Code',
            description: `Currently working as a large model algorithm engineer, writing about problems and solutions I encounter.
ps: Sorry, my articles are very AI-heavy, please read with caution! (I just don't want to write articles myself)
🐼 A novice adventurer, welcome to exchange!`
        },
        search: {
            placeholder: 'Search articles...',
            noResults: 'No results found',
            searching: 'Searching...'
        }
    }
}

export type Locale = keyof typeof i18n
export type I18nKey = typeof i18n[Locale] 