interface PostContent {
    zh: string
    en: string
}

export interface Post {
    title: {
        zh: string
        en: string
    }
    date: string
    author: string
    content: PostContent
    coverImage?: string
    tags?: string[]
    slug: string
} 