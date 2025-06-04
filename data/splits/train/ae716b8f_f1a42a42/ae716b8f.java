import java.io.*;
import java.util.*;

public class DisCenJump {

	public static void main(String[] args) throws IOException {

		DisCenJump obj = new DisCenJump();

		obj.doStuff();

	}

	int bslo(int v) {
		int l = 0, r = lo.size()-1;
		while (l < r) {
			int m = (l+r)/2;
			if (v <= lo.get(m)[0]) {
				if (v > lo.get(m+1)[0]) return m;
				l = m+1;
			} else r = m;
		}
		return l;
	}
	int bshi(int v) {
		int l = 0, r = hi.size()-1;
		while (l < r) {
			int m = (l+r)/2;
			if (v >= hi.get(m)[0]) {
				if (v < hi.get(m+1)[0]) return m;
				l = m+1;
			} else r = m;
		}
		return l;
	}
	
	int[] nums;
	int[] dp;
	ArrayList<int[]> lo, hi;
	private void doStuff() throws IOException {

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		nums = new int[Integer.parseInt(br.readLine())];
		StringTokenizer st = new StringTokenizer(br.readLine());
		for (int i = 0; i < nums.length; i++) {
			nums[i] = Integer.parseInt(st.nextToken());
		}
		br.close();
		
		dp = new int[nums.length];
		lo = new ArrayList<>();
		hi = new ArrayList<>();
		lo.add(new int[] {nums[nums.length-1], nums.length-1});
		hi.add(new int[] {nums[nums.length-1], nums.length-1});
		for (int i = dp.length-2; i >= 0; i--) {
			dp[i] = dp[i+1]+1;
			int pos = bslo(nums[i]);
			for (int j = pos; j < lo.size(); j++) {
				dp[i] = Math.min(dp[i], dp[lo.get(j)[1]]+1);
			}
			pos = bshi(nums[i]);
			for (int j = pos; j < hi.size(); j++) {
				dp[i] = Math.min(dp[i], dp[hi.get(j)[1]]+1);
			}
			
			while (lo.size() > 0 && lo.get(lo.size()-1)[0] <= nums[i]) {
				lo.remove(lo.size()-1);
			}
			lo.add(new int[] {nums[i], i});
			
			while (hi.size() > 0 && hi.get(hi.size()-1)[0] >= nums[i]) {
				hi.remove(hi.size()-1);
			}
			hi.add(new int[] {nums[i], i});
		}
		
		System.out.println(dp[0]);

	}

}
