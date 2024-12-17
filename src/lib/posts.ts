import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { categoryMap } from '@/config/categories'

const postsDirectory = path.join(process.cwd(), 'posts')

interface Post {
    slug: string
    title: {
        zh: string
        en: string
    }
    date: string
    author: string
    content: {
        zh: string
        en: string
    }
    excerpt: {
        zh: string
        en: string
    }
    coverImage?: string
    coverImageAlt?: {
        zh: string
        en: string
    }
    category: string
    tags?: string[]
    status: 'published' | 'draft'
}

export async function getAllPosts(includeDrafts: boolean = false): Promise<Post[]> {
    const entries = fs.readdirSync(postsDirectory, { withFileTypes: true })
    const posts: Post[] = []

    for (const entry of entries) {
        if (entry.isDirectory()) {
            const categoryPath = path.join(postsDirectory, entry.name)
            const files = fs.readdirSync(categoryPath)

            for (const fileName of files) {
                if (fileName.endsWith('.md')) {
                    const post = getPost(fileName.replace(/\.md$/, ''))
                    if (includeDrafts || post.status === 'published') {
                        posts.push(post)
                    }
                }
            }
        }
    }

    return posts.sort((a, b) => (a.date < b.date ? 1 : -1))
}

export async function getPostBySlug(slug: string, includeDrafts: boolean = false) {
    try {
        const post = getPost(slug)
        if (includeDrafts || post.status === 'published') {
            return post
        }
        return null
    } catch (error) {
        console.error(`Error getting post ${slug}:`, error)
        return null
    }
}

export async function getCategories() {
    const entries = fs.readdirSync(postsDirectory, { withFileTypes: true })
    return entries
        .filter(entry => entry.isDirectory())
        .map(entry => ({
            key: entry.name,
            name: {
                zh: categoryMap[entry.name]?.zh || entry.name,
                en: categoryMap[entry.name]?.en || entry.name
            }
        }))
}

export function getPost(slug: string): Post {
    const fullPath = findPostFile(slug)
    const fileContents = fs.readFileSync(fullPath, 'utf8')
    const { data, content } = matter(fileContents)

    // 使用正则表达式匹配中英文内容
    const zhMatch = content.match(/<!-- Chinese Content -->([\s\S]*?)(?=<!-- English Content -->|$)/);
    const enMatch = content.match(/<!-- English Content -->([\s\S]*?)$/);

    const zhContent = zhMatch ? zhMatch[1].trim() : '';
    const enContent = enMatch ? enMatch[1].trim() : '';

    // 生成摘要
    const generateExcerpt = (text: string) => {
        return text.slice(0, 200).replace(/#+\s/g, '').replace(/\[.*?\]\(.*?\)/g, '').trim() + '...'
    }

    return {
        slug,
        title: {
            zh: typeof data.title === 'object' ? data.title.zh : data.title,
            en: typeof data.title === 'object' ? data.title.en : data.title
        },
        date: data.date,
        author: data.author,
        content: {
            zh: zhContent,
            en: enContent
        },
        excerpt: {
            zh: generateExcerpt(zhContent),
            en: generateExcerpt(enContent)
        },
        coverImage: data.coverImage,
        coverImageAlt: {
            zh: typeof data.coverImageAlt === 'object' ? data.coverImageAlt.zh : data.coverImageAlt,
            en: typeof data.coverImageAlt === 'object' ? data.coverImageAlt.en : data.coverImageAlt
        },
        category: data.category || findPostCategory(slug),
        tags: data.tags || [],
        status: data.status || 'published'
    }
}

function findPostFile(slug: string) {
    const categories = fs.readdirSync(postsDirectory)

    for (const category of categories) {
        const categoryPath = path.join(postsDirectory, category)
        if (fs.statSync(categoryPath).isDirectory()) {
            const filePath = path.join(categoryPath, `${slug}.md`)
            if (fs.existsSync(filePath)) {
                return filePath
            }
        }
    }

    throw new Error(`Post not found: ${slug}`)
}

function findPostCategory(slug: string): string {
    const categories = fs.readdirSync(postsDirectory)

    for (const category of categories) {
        const categoryPath = path.join(postsDirectory, category)
        if (fs.statSync(categoryPath).isDirectory()) {
            const filePath = path.join(categoryPath, `${slug}.md`)
            if (fs.existsSync(filePath)) {
                return category
            }
        }
    }

    return ''
} 