layout(location = 0) in vec3 aPos;

uniform uint drawID;

uniform mat4 projection;
uniform mat4 view;

out VertexData {
	flat uint drawID;
	vec3 WorldPos;
} vsOut;

void main() {
	vsOut.drawID = drawID;
    vsOut.WorldPos = aPos;

	mat4 rotView = mat4(mat3(view));
	vec4 clipPos = projection * rotView * vec4(vsOut.WorldPos, 1.0);
	gl_Position = clipPos.xyww;
}
