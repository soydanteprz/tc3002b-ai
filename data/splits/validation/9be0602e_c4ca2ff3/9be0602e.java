import java.io.*;
import java.util.*;

public class Codeforce {

    

    static int mod = (int) (1e9 + 7);

    

    

    

    

    

    
public static  int fun(int ch,int a[][],int tot[],int n){
        ArrayList<Integer>ar=new ArrayList<>();
        for(int i=0;i<n;i++){
            ar.add((a[i][ch]-(tot[i]-a[i][ch])));
        }
        Collections.sort(ar,Collections.reverseOrder());
        int cou=0;
        int ans=0;
        for(int i:ar){
            cou+=i;
            if(cou>0){
                ans++;
            }
            else
                break;
        }
return ans;
}
    public static void main(String arg[]) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine());
        StringBuilder sb = new StringBuilder();
        while (t-- > 0) {
               int n=Integer.parseInt(br.readLine());
               int a[][]=new int[n][5];
               int tot[]=new int[n];
               for(int i=0;i<n;i++){
                   String s=br.readLine();
                   tot[i]+=s.length();
                   for(int j=0;j<s.length();j++){
                       a[i][s.charAt(j)-'a']++;
                   }
               }
               int ans=0;
               for(int i=0;i<5;i++){
                   ans=Math.max(ans,fun(i,a,tot,n));
               }
               sb.append(ans+"\n");
                       }
        System.out.println(sb);
    }
}
