import streamlit as st
import requests
import time

# Streamlit准备页面
st.title("饮料 及 生活用品 分类项目")
st.write("这是一个多分类项目")

# 获取用户输入
text = st.text_input("请输入文本")

# 发送请求
url = 'http://127.0.0.1:8010/predict'

if st.button("基于BERT原生模型获取分类！"):
    if not text.strip():
        st.error("请输入有效的文本内容")
    else:
        start_time = time.time()
        try:
            # 发送POST请求
            response = requests.post(url, json={'text': text})

            # 检查HTTP状态码
            if response.status_code == 200:
                result = response.json()

                # 检查API返回的成功标志
                if result.get('success'):
                    # 正确获取预测结果字段
                    prediction = result.get('prediction')
                    confidence = result.get('confidence')
                    input_text = result.get('input_text')

                    st.success(f"预测成功！")
                    st.write(f"输入文本：{input_text}")
                    st.write(f"预测结果：{prediction}")
                    st.write(f"置信度：{confidence}")

                    # 计算耗时
                    estimated_time = (time.time() - start_time) * 1000
                    st.write(f"耗时：{estimated_time:.2f} ms")
                else:
                    st.error(f"API返回错误：{result.get('error', '未知错误')}")
            else:
                st.error(f"HTTP请求失败，状态码：{response.status_code}")
                st.error(f"响应内容：{response.text}")

        except requests.exceptions.ConnectionError:
            st.error("无法连接到后端API服务，请确保Flask服务器正在运行（http://127.0.0.1:8010）")
        except requests.exceptions.Timeout:
            st.error("请求超时，请稍后重试")
        except KeyError as e:
            st.error(f"响应数据格式错误，缺少必要字段：{e}")
            st.write("实际响应内容：", response.json() if 'response' in locals() else "无响应")
        except Exception as e:
            st.error(f"发生错误：{str(e)}")
            import traceback

            st.write("详细错误信息：", traceback.format_exc())
