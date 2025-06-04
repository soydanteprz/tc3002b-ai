import java.util.*;

public class CodeForces1525C{
	public static void main(String[] args) {
		Scanner sc=new Scanner(System.in);
		ArrayList<Integer> o=new ArrayList<Integer>(), e=new ArrayList<Integer>();
		int n = sc.nextInt(),dp[][]=new int[n+1][n+1];
		for(int i=1;i<=n;i++){
			int x=sc.nextInt();
			if(x==1)o.add(i);
			else e.add(i);
		}
		for(int i=1;i<=o.size();i++){
			dp[i][i]=dp[i-1][i-1]+Math.abs(o.get(i-1)-e.get(i-1));
			for(int j=i+1;j<=e.size();j++)
				dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(o.get(i-1)-e.get(j-1)));
		}
		System.out.println(dp[o.size()][e.size()]);
	}
}
