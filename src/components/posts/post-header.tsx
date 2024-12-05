import { Button } from "@/components/ui/button"
import { ChevronLeft } from "lucide-react"
import Link from "next/link"

export function PostHeader() {
    return (
        <div className="sticky top-[64px] z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center">
                <Link href="/posts">
                    <Button variant="ghost" size="sm" className="gap-2">
                        <ChevronLeft className="h-4 w-4" />
                        返回文章列表
                    </Button>
                </Link>
            </div>
        </div>
    )
} 