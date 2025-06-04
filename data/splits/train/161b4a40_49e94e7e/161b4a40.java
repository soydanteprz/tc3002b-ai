import java.util.*;
import java.io.*;

public class Main{
    static final Random random=new Random();
    static long mod=1000000007L;
    static HashMap<String,Integer>map=new HashMap<>();

    
    
    public static void main(String[] args) {
        try {
            FastReader in=new FastReader();
            FastWriter out = new FastWriter();
            int testCases=in.nextInt();
            //int testCases=1;
            while(testCases-- > 0){
                solve(in);
            }
            out.close();
        } catch (Exception e) {
            return;
        }
    }
    public static void solve( FastReader in){
        int n=in.nextInt();
        String s=in.next();
        String t=in.next();
        //int k=in.nextInt();
        //long y=in.nextInt();
        //long n=in.nextLong();
        //int k=in.nextInt();
        //long k=in.nextLong();
        StringBuilder res=new StringBuilder();
        char[] s1=s.toCharArray();
        char[] t1=t.toCharArray();
        int ans=n+2;
        int[] cnt={0,0};
        for(int i=0;i<n;i++){
            if(s1[i]=='0' && t1[i]=='1'){
                cnt[0]++;
            }
            if(s1[i]=='1' && t1[i]=='0'){
                cnt[1]++;
            }
        }
        if(cnt[0]==cnt[1])ans=Math.min(ans,cnt[0]+cnt[1]);
        cnt[0]=cnt[1]=0;
        for(int i=0;i<n;i++){
            if(s1[i]=='0' && t1[i]=='0'){
                cnt[0]++;
            }
            if(s1[i]=='1' && t1[i]=='1'){
                cnt[1]++;
            }
        }
        if(cnt[1]==cnt[0]+1){
            ans=Math.min(ans,cnt[0]+cnt[1]);
        }
        if(ans>n){
            res.append("-1");
        }
        else{
            res.append(""+ans);
        }
        //int ans=x.size()+y.size();
        //res.append(""+"Yes");
        //res.append(""+"");
        System.out.println(res.toString());
    }
    
    

    
    

    

    
    static < E > void print(E res)
    {
        System.out.println(res);
    }
    
}
