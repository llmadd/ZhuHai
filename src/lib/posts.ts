import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const postsDirectory = path.join(process.cwd(), 'posts')

// 定义文章类型接口
interface Post {
    slug: string;
    title: string;
    date: string;
    author: string;
    category: string;
    excerpt: string;
    content: string;
    coverImage?: string;
    tags?: string[];
    status: 'published' | 'draft';
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
                    const fullPath = path.join(categoryPath, fileName)
                    const fileContents = fs.readFileSync(fullPath, 'utf8')
                    const { data, content } = matter(fileContents)

                    // 如果文章没有明确设置状态，默认为已发布
                    const status = data.status || 'published'

                    // 只有当 includeDrafts 为 true 时才包含草稿，或者文章状态为 published
                    if (includeDrafts || status === 'published') {
                        posts.push({
                            slug: fileName.replace(/\.md$/, ''),
                            title: data.title || '无标题',
                            date: data.date || new Date().toISOString(),
                            author: data.author || '匿名',
                            category: entry.name,
                            excerpt: data.excerpt || content.slice(0, 200) + '...',
                            content,
                            coverImage: data.coverImage,
                            tags: data.tags || [],
                            status
                        })
                    }
                }
            }
        }
    }

    return posts.sort((a, b) => (a.date < b.date ? 1 : -1))
}

export async function getPostBySlug(slug: string, includeDrafts: boolean = false) {
    const posts = await getAllPosts(includeDrafts)
    return posts.find(post => post.slug === slug)
}

export async function getCategories() {
    const entries = fs.readdirSync(postsDirectory, { withFileTypes: true })
    return entries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
} 