import java.util.*;
public class MyClass {
    public static void main(String args[]) {
     Scanner s=new Scanner(System.in);
     int n=s.nextInt();
     int a[]=new int[n];
     ArrayList<Integer> lt1=new ArrayList<>();
     ArrayList<Integer> lt0=new ArrayList<>();
     for(int i=0;i<n;i++)
     {
         int l=s.nextInt();
         if(l==0)
         lt0.add(i+1);
         else
         lt1.add(i+1);
     }
     int dp[][]=new int[lt1.size()+1][lt0.size()+1];
     for(int i=1;i<=lt1.size();i++)
     {
         dp[i][i]=dp[i-1][i-1]+Math.abs(lt0.get(i-1)-lt1.get(i-1));
         for(int j=i+1;j<=lt0.size();j++)
         {
             dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(lt1.get(i-1)-lt0.get(j-1)));
         }
     }
     System.out.println(dp[lt1.size()][lt0.size()]);
     
    }
}
