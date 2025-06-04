import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.ArrayList;
public class Armchairs {
	static ArrayList<Integer> one;
	static ArrayList<Integer> zero;
	public static void main(String args[]) {
		FastScanner fs=new FastScanner();
		int n=fs.nextInt();
		int []arr=new int[n];
		one=new ArrayList<>();
		zero=new ArrayList<>();
		for(int i=0;i<arr.length;i++) {
			arr[i]=fs.nextInt();
			if(arr[i]==1)
				one.add(i);
			else zero.add(i);
		}
		int [][]dp=new int[arr.length][arr.length];
		for(int i=0;i<arr.length;i++)
			Arrays.fill(dp[i], -1);
		System.out.println(dfs(dp,0,0));
	}
	public static int dfs(int [][]dp,int i,int j) {
		//System.out.println(i+"  "+j);
		if(i>=one.size())
			return 0;
		if(j>=zero.size())
			return (int)(1e9);
		if(dp[i][j]!=-1)
			return dp[i][j];
		dp[i][j]=Math.min(Math.abs(one.get(i)-zero.get(j))+dfs(dp,i+1,j+1), dfs(dp,i,j+1));
		return dp[i][j];
	}
	

static final Random random=new Random();
	
	
	
}
