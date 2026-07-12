if uploaded_file is not None:
    pil_image_rgb = Image.open(uploaded_file).convert("RGB")
    pil_image = pil_image_rgb.convert("L")
    image_tensor = transform(pil_image)

    is_valid, reasons, diag_info = analyze_image_validity(pil_image_rgb)

    start_time = time.time()
    probs = predict(model, image_tensor, device)
    inference_ms = (time.time() - start_time) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_confidence = float(probs[pred_idx])
    low_confidence = pred_confidence < 0.55

    if not is_valid or low_confidence:
        st.error("⚠️ This image does not appear to be a valid T1-weighted MRI scan.")
        for reason in reasons:
            st.write(f"- {reason}")
        if low_confidence:
            st.write(f"- Model confidence is low ({pred_confidence:.1%}), suggesting the input doesn't resemble the training distribution.")

        with st.expander("Diagnostic details"):
            st.json(diag_info)

        proceed = st.checkbox(
            "I understand this image failed validity checks and want to see the raw model output anyway (research use only).",
            key=f"override_{uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name}"
        )
        if not proceed:
            st.stop()

    meta = STAGE_META[pred_class]
    laai_score = pred_confidence / (1 + (inference_ms / 1000))

    st.session_state.case_log.insert(0, {
        "Patient ID": patient_id or "—",
        "Scan ID": scan_id or "—",
        "Date": scan_date.strftime("%Y-%m-%d"),
        "Result": pred_class,
        "Confidence": f"{pred_confidence:.1%}",
        "Time (ms)": f"{inference_ms:.1f}",
    })

    st.markdown(f"""
    <div class="verdict-card" style="background:{meta['bg']}; border-left-color:{meta['color']};">
        <span class="verdict-badge" style="background:{meta['color']};">{meta['badge']}</span>
        <div class="verdict-title" style="color:{meta['color']};">{pred_class}</div>
        <div class="verdict-sub">{meta['summary']}</div>
        <div class="verdict-action" style="color:{meta['color']};">➤ {meta['action']}</div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="label">Confidence</div><div class="value">{pred_confidence:.1%}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="label">Inference Time</div><div class="value">{inference_ms:.1f} ms</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="label">LAAI Score</div><div class="value">{laai_score:.3f}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="label">Model</div><div class="value" style="font-size:15px;">CNN-CapsNet</div></div>', unsafe_allow_html=True)

    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Breakdown", "🔥 Grad-CAM++", "🟡 LIME", "⚙️ Technical Details"])

    with tab1:
        col_img, col_chart = st.columns([1, 1.4])
        with col_img:
            st.image(pil_image, caption="Uploaded MRI", use_container_width=True)
        with col_chart:
            st.write("**Class probabilities**")
            for i, cname in enumerate(class_names):
                st.write(f"{cname}")
                st.progress(float(probs[i]), text=f"{probs[i]:.1%}")

    with tab2:
        st.caption("Highlights the brain regions the CNN backbone weighted most heavily for this prediction.")
        with st.spinner("Generating Grad-CAM++ heatmap..."):
            gradcam_overlay = generate_gradcam(model, image_tensor, device)
        c1, c2 = st.columns(2)
        c1.image(pil_image, caption="Original MRI", use_container_width=True)
        c2.image(gradcam_overlay, caption="Grad-CAM++ Overlay", use_container_width=True)

    with tab3:
        st.caption("Yellow contours mark the superpixel regions that most positively influenced the predicted class.")
        with st.spinner("Generating LIME explanation (~10–20s)..."):
            lime_overlay = generate_lime(model, image_tensor, device)
        c1, c2 = st.columns(2)
        c1.image(pil_image, caption="Original MRI", use_container_width=True)
        c2.image(lime_overlay, caption="LIME Explanation", use_container_width=True)

    with tab4:
        st.write("**Raw model output**")
        st.json({cname: float(f"{probs[i]:.6f}") for i, cname in enumerate(class_names)})
        st.write("**System info**")
        st.json({
            "device": str(device),
            "input_size": f"{img_size}x{img_size}",
            "inference_time_ms": round(inference_ms, 3),
            "laai_score": round(laai_score, 4),
        })

else:
    st.info("👆 Upload an MRI scan above to begin screening.")
