import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.*;
import java.io.*;
public class Main {
    public static void main(String args[])throws Exception{
        Input sc=new Input();
        StringBuilder sb=new StringBuilder();
        int t=sc.readInt();
        for(int f=0;f<t;f++){
            int n=sc.readInt();
            String str[]=new String[n];
            for(int i=0;i<n;i++){
                str[i]=sc.readString();
            }
            char ch[]={'a','b','c','d','e'};
            int ans=0;
            for(int i=0;i<5;i++){
                char c=ch[i];
                ArrayList<pair> lst=new ArrayList<>();
                int a[]=new int[n];
                for(int j=0;j<n;j++)
                {
                    int count=0;
                    for(int k=0;k<str[j].length();k++){
                        if(str[j].charAt(k)==c)
                            count++;
                    }
                    lst.add(new pair(count,(str[j].length()-count)));
                }
                Collections.sort(lst);
                int s1=0;int s2=0;
//                for(int j=0;j<lst.size();j++){
//                    System.out.println(lst.get(j).a+" "+lst.get(j).b);
//                }
//                System.out.println("--------");
                int co=0;
                for(int j=0;j<lst.size();j++){
                    int v1=lst.get(j).a;
                    int v2=lst.get(j).b;
                    if((s1+v1)>(s2+v2)){
                        s1+=v1;
                        s2+=v2;
                        co++;
                    }
                    //System.out.println(lst.get(j).a+" "+lst.get(j).b);
                }
                ans=Math.max(co,ans);

            }
            // count all 'a' in each string
            sb.append(ans+"\n");
        }
        System.out.print(sb);
    }
}
class pair implements Comparable<pair>{
    int a;
    int b;
    pair(int a,int b){
        this.a=a;
        this.b=b;
    }

    
}

