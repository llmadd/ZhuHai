'use client'

import { useState, useRef } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { Share2, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import * as htmlToImage from 'html-to-image'
import { useLocale } from "@/contexts/locale-context"
import { i18n } from "@/config/i18n"

interface ShareCardProps {
    title: string
    url: string
}

export function ShareCard({ title, url }: ShareCardProps) {
    const [isGenerating, setIsGenerating] = useState(false)
    const cardRef = useRef<HTMLDivElement>(null)

    const { locale } = useLocale()
    const t = i18n[locale]

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
                <span className="font-medium">{t.post.share}</span>
            </div>

            <div
                ref={cardRef}
                className="bg-background p-6 rounded-lg border"
            >
                <h2 className="text-xl font-bold mb-4">{title}</h2>
                <div className="flex items-center justify-between">
                    <div className="text-sm text-muted-foreground">
                        <p>{t.post.scanQRCode}</p>
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
                {isGenerating ? t.post.generating : t.post.downloadShareImage}
            </Button>
        </div>
    )
} 