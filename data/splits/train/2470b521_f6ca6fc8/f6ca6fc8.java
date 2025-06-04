import java.util.*;
import java.io.*;

public class D {
	public static void main(String[] args) {
		Scanner sc=new Scanner(System.in);
		int n = sc.nextInt();
		ArrayList<Integer>  o=new ArrayList<Integer>();
		ArrayList<Integer>  e=new ArrayList<Integer>();
		for(int i=1;i<=n;i++){
			int x=sc.nextInt();
			if(x==1)o.add(i);
			else e.add(i);
		}
		int dp[][]=new int[o.size()+1][e.size()+1];
		for(int i=1;i<=o.size();i++){
			dp[i][i]=dp[i-1][i-1]+Math.abs(o.get(i-1)-e.get(i-1));
			for(int j=i+1;j<=e.size();j++){
				dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(o.get(i-1)-e.get(j-1)));
			}
		}
		System.out.println(dp[o.size()][e.size()]);
	}
}
