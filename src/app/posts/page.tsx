import { getAllPosts, getCategories } from "@/lib/posts"
import { PostList } from "@/components/index/post-list"

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