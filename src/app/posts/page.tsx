import { Metadata } from "next"
import { siteConfig } from "@/config/site"
import { getAllPosts, getCategories } from "@/lib/posts"
import { PostList } from "@/components/index/post-list"

export const metadata: Metadata = {
    title: "博客文章",
    description: "浏览所有文章，包括技术分享和生活感悟",
    openGraph: {
        title: "博客文章 | " + siteConfig.name,
        description: "浏览所有文章，包括技术分享和生活感悟",
        url: `${siteConfig.url}/posts`,
    },
}

export default async function PostsPage() {
    const [posts, categories] = await Promise.all([
        getAllPosts(false),
        getCategories()
    ])

    return (
        <div className="container">
            <PostList posts={posts} categories={categories} />
        </div>
    )
} 