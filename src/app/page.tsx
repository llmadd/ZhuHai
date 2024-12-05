import { PostList } from "@/components/index/post-list"
import { ProfileCard } from "@/components/index/profile-card"
import { getAllPosts, getCategories } from "@/lib/posts"

export default async function Home() {
  const [posts, categories] = await Promise.all([
    getAllPosts(false),
    getCategories()
  ])

  return (
    <main className="container">
      <ProfileCard />
      <PostList posts={posts} categories={categories} />
    </main>
  )
}
