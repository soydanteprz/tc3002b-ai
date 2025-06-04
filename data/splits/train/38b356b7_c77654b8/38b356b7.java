import java.awt.*;
import java.io.*;
import java.util.*;

public class Main {
	static BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
	static StringBuilder str = new StringBuilder();
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		int t = Integer.parseInt(reader.readLine());
		while (t-- > 0) {
			solve();
		}
		printRes();
	}
	
	public static void printRes() {
		
		System.out.println(str);
	}

	public static void solve() throws IOException {
		int n = Integer.parseInt(reader.readLine());
		String[] s = reader.readLine().split(" ");
		int[] nums = new int[n];
		
		for (int i = 0; i < n; i++) {
			nums[i] = Integer.parseInt(s[i]);
		}
		
		int i = 0;
		
		if (n % 2 == 1) {
			if (nums[1] + nums[2] != 0) {
				str.append(-(nums[1] + nums[2]) + " " + nums[0] + " " + nums[0] + " ");
			} else if (nums[0] + nums[2] != 0) {
				str.append(nums[1] + " " + -(nums[0] + nums[2]) + " " + nums[1] + " ");
			} else {
				str.append(nums[2] + " " + nums[2] + " " + -(nums[0] + nums[1]) + " ");
			}
			i = 3;
		}
		for (; i < n; i+=2) {
			if (nums[i] > 0 && nums[i+1] > 0) {
				str.append(nums[i+1] + " " + (-nums[i]) + " ");
			} else if (nums[i] < 0 && nums[i+1] < 0) {
				str.append(nums[i+1] + " " + Math.abs(nums[i]) + " ");
			} else {
				str.append(Math.abs(nums[i+1]) + " " + Math.abs(nums[i]) + " ");
			}
		}
		str.append("\n");
	}
}
