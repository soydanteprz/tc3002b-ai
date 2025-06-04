import java.util.*;

public class main {
    public static final Scanner sc = new Scanner(System.in);

    public static void solve(){
        int n=sc.nextInt();
        int[][] a=new int[5][n];
        for(int i=0;i<n;i++){
            String x=sc.next();
            for(int j=0;j<x.length();j++){
                if(x.charAt(j)=='a') a[0][i]++;
                else a[0][i]--;
                if(x.charAt(j)=='b') a[1][i]++;
                else a[1][i]--;
                if(x.charAt(j)=='c') a[2][i]++;
                else a[2][i]--;
                if(x.charAt(j)=='d') a[3][i]++;
                else a[3][i]--;
                if(x.charAt(j)=='e') a[4][i]++;
                else a[4][i]--;
            }
        }
        for(int j=0;j<5;j++) Arrays.sort(a[j]);
        int ans=0;
        for(int j=0;j<5;j++){
            int tmp=0,cnt=0;
            for(int i=n-1;i>=0;i--){
                tmp+=a[j][i];
                if(tmp<=0) break;
                else cnt++;
            }
            ans=Math.max(ans,cnt);
        }
        System.out.println(ans);
    }
    public static void main(String[] args) {
        int T=1;
        T=sc.nextInt();
        while(--T>=0) solve();
    }
}
