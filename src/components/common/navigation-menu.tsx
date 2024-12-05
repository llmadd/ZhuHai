import Link from "next/link"
import Image from "next/image"
import { SearchDialog } from "./search-dialog"
import { ThemeToggle } from "./theme-toggle"

interface NavigationMenuProps {
    className?: string;
}

export function NavigationMenu({ className }: NavigationMenuProps) {
    return (
        <nav className={className}>
            <div className="container flex h-16 items-center">
                <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-6">
                        <Link href="/" className="flex items-center gap-2">
                            <Image
                                src="/logo.png"
                                alt="Logo"
                                width={32}
                                height={32}
                                className="rounded-lg"
                            />
                            <span className="text-xl font-bold">ZhuHai.Fun</span>
                        </Link>
                    </div>
                    <div className="flex items-center gap-4">
                        <SearchDialog />
                        <Link href="/posts" className="hover:text-primary">
                            分类
                        </Link>
                        <ThemeToggle />
                    </div>
                </div>
            </div>
        </nav>
    )
} 