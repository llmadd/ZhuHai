'use client'

import { Button } from "@/components/ui/button"
import Image from "next/image"
import { Github, Twitter, MessageCircle } from "lucide-react"
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import Link from "next/link"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

export function ProfileCard() {
    const { locale } = useLocale()
    const t = i18n[locale]

    return (
        <section className="min-h-[calc(100vh-80px)] flex items-center justify-center">
            <div className="text-center">
                <div className="relative w-32 h-32 mx-auto mb-6">
                    <Image
                        src="/logo.png"
                        alt="头像"
                        fill
                        className="object-cover rounded-full"
                        priority
                    />
                </div>
                <h1 className="text-4xl font-bold mb-4">{t.profile.name}</h1>
                <p className="text-xl text-muted-foreground mb-4 max-w-2xl mx-auto flex items-center justify-center gap-2">
                    <span>{t.profile.location}</span>
                    <span>·</span>
                    <span>{t.profile.email}</span>
                </p>
                <div className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto whitespace-pre-line">
                    {t.profile.description}
                </div>
                <div className="flex items-center justify-center gap-4 mb-8">
                    <TooltipProvider>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button variant="outline" size="icon">
                                    <Link href="https://github.com/llmadd" target="_blank">
                                        <Github className="h-5 w-5" />
                                    </Link>
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>Github</p>
                            </TooltipContent>
                        </Tooltip>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button variant="outline" size="icon">
                                    <Link href="https://x.com/qichenAi" target="_blank">
                                        <Twitter className="h-5 w-5" />
                                    </Link>
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>X (Twitter)</p>
                            </TooltipContent>
                        </Tooltip>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Dialog>
                                    <DialogTitle className="sr-only">{t.profile.wechatQRCode}</DialogTitle>
                                    <DialogTrigger asChild>
                                        <Button variant="outline" size="icon">
                                            <MessageCircle className="h-5 w-5" />
                                        </Button>
                                    </DialogTrigger>
                                    <DialogContent>
                                        <div className="relative w-full h-[300px]">
                                            <Image
                                                src="/wechat.jpg"
                                                alt={t.profile.wechatQRCode}
                                                fill
                                                className="object-contain"
                                            />
                                        </div>
                                    </DialogContent>
                                </Dialog>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>{t.profile.wechat}</p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                </div>
                <Button size="lg" asChild>
                    <Link href="/posts">{t.profile.viewBlog}</Link>
                </Button>
            </div>
        </section>
    )
} 