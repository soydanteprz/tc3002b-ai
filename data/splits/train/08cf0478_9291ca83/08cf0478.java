import java.util.*;
import java.io.*;

public class Main {

  

  public static void main(final String[] args) {
    FastScanner sc = new FastScanner();

	int t = sc.nextInt();
	while(t-- > 0) {
		int n = sc.nextInt();
		int[][] arr = new int[5][n];
		
		for(int i = 0; i < n; i++) {
			char[] s = sc.next().toCharArray();
			int[] cnt = new int[5];
			for(int j = 0; j < s.length; j++) {
				cnt[s[j]-'a']++;
			}
			for(int j = 0; j < 5; j++)
				arr[j][i] = cnt[j]-(s.length-cnt[j]);
		}
		
		int ans = 0;
		for(int i = 0; i < 5; i++) {
			Arrays.sort(arr[i]);
			int maxSum = 0;
			int words = 0;
			for(int j = arr[i].length-1; j >=0; j--) {
				maxSum += arr[i][j];
				if(maxSum > 0) words++;
			}
			
			ans = Math.max(ans, words);
		}
		
		System.out.println(ans);
	}

  } 

  
}
