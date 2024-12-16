'use client'

import { useState, useRef } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { Share2, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import * as htmlToImage from 'html-to-image'

interface ShareCardProps {
    title: string
    url: string
}

export function ShareCard({ title, url }: ShareCardProps) {
    const [isGenerating, setIsGenerating] = useState(false)
    const cardRef = useRef<HTMLDivElement>(null)

    const generateImage = async () => {
        if (!cardRef.current) return
        setIsGenerating(true)
        try {
            const dataUrl = await htmlToImage.toPng(cardRef.current)
            const link = document.createElement('a')
            link.download = `${title}-分享图.png`
            link.href = dataUrl
            link.click()
        } catch (error) {
            console.error('生成图片失败:', error)
        }
        setIsGenerating(false)
    }

    return (
        <div className="mt-8">
            <div className="flex items-center gap-2 mb-4">
                <Share2 className="w-5 h-5" />
                <span className="font-medium">分享文章</span>
            </div>

            <div
                ref={cardRef}
                className="bg-background p-6 rounded-lg border"
            >
                <h2 className="text-xl font-bold mb-4">{title}</h2>
                <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                        <p>扫描二维码查看文章</p>
                        <p className="mt-1">ZhuHai.Fun</p>
                    </div>
                    <QRCodeSVG
                        value={url}
                        size={120}
                        level="H"
                        className="bg-white p-2 rounded"
                    />
                </div>
            </div>

            <Button
                className="mt-4"
                onClick={generateImage}
                disabled={isGenerating}
            >
                <Download className="w-4 h-4 mr-2" />
                {isGenerating ? '生成中...' : '下载分享图'}
            </Button>
        </div>
    )
} 