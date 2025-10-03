layout(location = 0) in vec3 aPos;

uniform uint DrawID;

uniform mat4 projection;
uniform mat4 view;

out VertexData {
	flat uint DrawID;
	vec3 WorldPos;
} vsOut;

void main() {
	vsOut.DrawID = DrawID;
    vsOut.WorldPos = aPos;

	mat4 rotationMatrix = mat4(mat3(view));
	vec4 clipPos = projection * rotationMatrix * vec4(vsOut.WorldPos, 1.0);
	gl_Position = clipPos.xyww;
}
