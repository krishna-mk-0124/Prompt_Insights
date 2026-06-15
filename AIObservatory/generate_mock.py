import random
import os

templates = [
    # English
    "How do I reset the password for my Amex corporate card?",
    "Can you help me pull the latest transaction report for merchant {merchant}?",
    "What is the current exchange rate for USD to {currency} for international settlements?",
    "Please explain the new American Express platinum benefits for Q3.",
    "I need to file a dispute for a charge of {amount} on my corporate account.",
    "Where can I find the compliance guidelines for cross-border payments?",
    "How do I process a refund for a customer whose card ends in {digits}?",
    "Is the API endpoint for real-time authorizations down today?",
    "I am getting an error code 403 when trying to access the internal Amex risk dashboard.",
    "What are the limits for Centurion lounge access this year?",
    
    # Spanish
    "¿Cómo restablezco la contraseña de mi tarjeta corporativa Amex?",
    "¿Puedes ayudarme a obtener el último informe de transacciones para el comerciante {merchant}?",
    "¿Cuál es el tipo de cambio actual de USD a {currency} para liquidaciones internacionales?",
    "Por favor, explique los nuevos beneficios de American Express Platinum para el tercer trimestre.",
    "Necesito presentar una disputa por un cargo de {amount} en mi cuenta corporativa.",
    "¿Dónde puedo encontrar las pautas de cumplimiento para pagos transfronterizos?",
    "¿Cómo proceso un reembolso para un cliente cuya tarjeta termina en {digits}?",
    "¿Está caído hoy el endpoint de la API para autorizaciones en tiempo real?",
    "Obtengo un código de error 403 al intentar acceder al panel de riesgo interno de Amex.",
    "¿Cuáles son los límites para el acceso a la sala VIP Centurion este año?",
    
    # Chinese (Simplified)
    "我该如何重置我的美国运通公司卡的密码？",
    "你能帮我提取商户 {merchant} 的最新交易报告吗？",
    "目前美元兑换 {currency} 的国际结算汇率是多少？",
    "请解释一下第三季度新的美国运通白金卡福利。",
    "我需要对我的公司账户上 {amount} 的扣款提出争议。",
    "我在哪里可以找到跨境支付的合规指南？",
    "我该如何为卡号以 {digits} 结尾的客户办理退款？",
    "实时授权的 API 端点今天宕机了吗？",
    "尝试访问内部运通风险仪表板时收到错误代码 403。",
    "今年百夫长休息室的准入限制是什么？",

    # Japanese
    "アメックスのコーポレートカードのパスワードをリセットするにはどうすればよいですか？",
    "加盟店 {merchant} の最新の取引レポートを抽出するのを手伝ってくれませんか？",
    "国際決済のための米ドルから {currency} への現在の為替レートはいくらですか？",
    "第3四半期の新しいアメリカン・エキスプレス・プラチナの特典について説明してください。",
    "法人アカウントでの {amount} の請求について異議を申し立てる必要があります。",
    "クロスボーダー決済のコンプライアンスガイドラインはどこにありますか？",
    "カードの末尾が {digits} のお客様への返金はどのように処理すればよいですか？",
    "リアルタイム承認用のAPIエンドポイントは今日ダウンしていますか？",
    "社内のアメックスリスクダッシュボードにアクセスしようとすると、エラーコード403が表示されます。",
    "今年のセンチュリオン・ラウンジの利用制限はどうなっていますか？"
]

merchants = ["Amazon", "Walmart", "Target", "Starbucks", "Delta", "Marriott", "Uber", "Apple", "Google", "Microsoft"]
currencies = ["EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD", "SGD", "MXN", "BRL"]

os.makedirs("data", exist_ok=True)
with open("data/prompt_sample", "w", encoding="utf-8") as f:
    for i in range(150000):
        template = random.choice(templates)
        amount = f"${random.randint(10, 5000)}.{random.randint(0, 99):02d}"
        digits = f"{random.randint(1000, 9999)}"
        merchant = random.choice(merchants)
        currency = random.choice(currencies)
        
        prompt = template.replace("{merchant}", merchant).replace("{currency}", currency).replace("{amount}", amount).replace("{digits}", digits)
        
        # Add a bit of random noise to make them unique
        noise = random.choice(["", " Thanks.", " Urgent.", " Help.", " ASAP.", " Please advise.", " Thanks in advance.", " Pls."])
        prompt += noise
        
        f.write(prompt + "\n")

print("Generated 150,000 mock prompts in data/prompt_sample")
