import java.util.*;
import java.io.*;
public class stoneheaps {
	public static void main(String[] args) throws Exception{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		int t = Integer.parseInt(br.readLine());
		for(int i = 0; i < t; i ++) {
			int n = Integer.parseInt(br.readLine());
			String[] tokens = br.readLine().split(" ");
			int[] arr = new int[n];
			for(int j = 0; j < n; j++) {
				arr[j] = Integer.parseInt(tokens[j]);
			}
			
			int l = 1;
			int r = 1000000000;
			while(l < r) {
				int mid  = l + (r-l+1)/2;
				if(check(arr, mid)) {
					l = mid;
				}
				else {
					r = mid-1;
				}
			}
			System.out.println(l);
		}
	}
	
	static boolean check(int[] arr, int x) {
		int[] changed = new int[arr.length];
		
		for(int i = 0; i < arr.length; i ++) {
			changed[i] = arr[i];
		}
		
		int n = arr.length;
		for(int i = n-1; i >= 0; i--) {
			if(changed[i] < x) {
				return false;
			}
			else if(i > 1){
				int change = Math.min(changed[i]-x, arr[i])/3;
				changed[i-1] = changed[i-1]+change;
				changed[i-2] = changed[i-2] +change*2;
			}
		}
		return true;
	}
}
